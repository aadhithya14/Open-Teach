/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using System;
using System.Collections.Generic;

namespace Oculus.Interaction
{
    /// <summary>
    /// MAction can be used in place of Action. This allows
    /// for interfaces with Actions of generic covariant types
    /// to be subscribed to by multiple types of delegates.
    /// </summary>
    public interface MAction<out T>
    {
        event Action<T> Action;
    }

    /// <summary>
    /// Classes that implement an interface that has MActions
    /// can use MultiAction as their MAction implementation to
    /// allow for multiple types of delegates to subscribe to the
    /// generic type.
    /// </summary>
    public class MultiAction<T> : MAction<T>
    {
        protected HashSet<Action<T>> actions = new HashSet<Action<T>>();

        public event Action<T> Action
        {
            add
            {
                actions.Add(value);
            }
            remove
            {
                actions.Remove(value);
            }
        }

        public void Invoke(T t)
        {
            foreach (Action<T> action in actions)
            {
                action(t);
            }
        }
    }
}
